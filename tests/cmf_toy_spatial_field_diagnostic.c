/* 같은 구간별 상수 물질의 수송 공간 민감도. 생산 상태는 변경하지 않는다. */
#define main unused_bf_diagnostic_main
#include "cmf_toy_bf_iterate_diagnostic.c"
#undef main
#include "cmf_exact_multigpu.h"
#include <float.h>

extern void w5_cont_selftest_arm_capture(double *,double *,double *,double *,double *,double *,size_t);
extern int w5_cont_selftest_capture_shape(int *,int *);

typedef struct { int ns,parent[256],center[256]; double ri[256],ro[256]; } SplitGrid;
static int split_grid(SplitGrid *g,int ns,const double *ri,const double *ro,int first,int last){
    if(!g||!ri||!ro||ns<1||ns>256||first<0||last<first||last>=ns||ns+2*(last-first+1)>256)return -1;
    memset(g,0,sizeof(*g));
    for(int s=0;s<ns;s++){
        if(!isfinite(ri[s])||!isfinite(ro[s])||ri[s]<0||ro[s]<=ri[s]||(s&&ri[s]!=ro[s-1]))return -1;
        int count=s>=first&&s<=last?3:1,base=g->ns;
        long double volume=0;
        for(int k=0;k<count;k++){
            int d=g->ns++;
            g->parent[d]=s;
            g->ri[d]=k?g->ro[d-1]:ri[s];
            g->ro[d]=k+1==count?ro[s]:ri[s]+(ro[s]-ri[s])*(k+1)/count;
            if(!(g->ro[d]>g->ri[d]))return -1;
            long double a=g->ri[d],b=g->ro[d];volume+=(b-a)*(b*b+b*a+a*a);
        }
        g->center[s]=base+count/2;
        double old=.5*(ri[s]+ro[s]),now=.5*(g->ri[g->center[s]]+g->ro[g->center[s]]);
        long double a=ri[s],b=ro[s],expect=(b-a)*(b*b+b*a+a*a);
        if(fabs(now-old)>4*DBL_EPSILON*fmax(fabs(old),fabs(now))||fabsl(volume/expect-1)>4e-15L)return -1;
    }
    return 0;
}
static double *replicate(const SplitGrid *g,int nb,const double *old){
    size_t cells=(size_t)g->ns*nb;double *out=malloc(cells*sizeof(*out));if(!out)return NULL;
    for(int s=0;s<g->ns;s++)memcpy(out+(size_t)s*nb,old+(size_t)g->parent[s]*nb,nb*sizeof(*out));
    return out;
}
static int snapshot(const CMFToyModel *m,double *j,const char *path,int iteration){
    CMFToyModel v=*m;v.nlte.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    v.nlte.radiation_field.field.generation.computed_generation=0;
    FILE *fp=fopen(path,"wx");if(!fp)return -1;
    int rc=fprintf(fp,"field_iteration,population_iteration,shell,Z,velocity_km_s,T_K,ne_cm3,III_II,channels,photo_per_II_s,recomb_per_III_s,collision_per_II_s,threebody_per_III_s,forward_cm3_s,reverse_cm3_s,signed_imbalance\n")<0;
    if(!rc)rc=evaluate(&v,j,v.nlte.nlte_level_populations,v.atom.ion_number_density,
        v.plasma.n_electron,v.plasma.T_e,iteration,iteration,fp);
    if(fclose(fp))rc=-1;return rc;
}
static int small_test(void){
    const int ns=4,nb=32;double ri[]={1e13,2e13,3e13,4e13},ro[]={2e13,3e13,4e13,5e13};
    double nu[32],chi[128],es[128],eta[128],j[128];SplitGrid g={0},bad={0};
    if(split_grid(&g,ns,ri,ro,1,2)||g.ns!=8||!split_grid(&bad,ns,ri,ro,-1,2))return -1;
    for(int b=0;b<nb;b++)nu[b]=3e15*exp((b+.5)*.001);
    for(int s=0;s<ns;s++)for(int b=0;b<nb;b++){
        int i=s*nb+b;chi[i]=(1+s)*1e-13;es[i]=.25*chi[i];
        double B=2*H_PLANCK*nu[b]*nu[b]*nu[b]/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(H_PLANCK*nu[b]/(K_BOLTZMANN*(9000+1000*s)));
        eta[i]=(chi[i]-es[i])*B;j[i]=B;
    }
    double *c=replicate(&g,nb,chi),*e=replicate(&g,nb,es),*f=replicate(&g,nb,eta),*cpu=replicate(&g,nb,j);
    double *gpu=replicate(&g,nb,j),*ce=calloc((size_t)g.ns*nb,sizeof(double)),*ge=calloc((size_t)g.ns*nb,sizeof(double));
    int rc=-1;double difference=0;
    if(!c||!e||!f||!cpu||!gpu||!ce||!ge)goto finish;
    for(int s=0;s<g.ns;s++)if(memcmp(c+(size_t)s*nb,chi+(size_t)g.parent[s]*nb,nb*sizeof(double)))goto finish;
    CMFExactReport cr={0};CMFMultiGPUReport gr={0};CMFMultiGPUEpochSchedule schedule={128,4096,32};
    if(cmf_exact_emissivity_solve(g.ns,nb,.001,nu,g.ri,g.ro,1e5,9000,1,c,e,f,cpu,ce,16,256,1e-10,
        CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)!=CMF_EXACT_OK)goto finish;
    if(cmf_exact_multigpu_emissivity_solve_epoch(g.ns,nb,.001,nu,g.ri,g.ro,1e5,9000,1,c,e,f,gpu,ge,
        16,&schedule,1,256,1e-10,&gr)!=CMF_MGPU_OK)goto finish;
    for(int i=0;i<g.ns*nb;i++){
        if(!(cpu[i]>0)||!isfinite(gpu[i]))goto finish;
        difference=fmax(difference,fabs(gpu[i]/cpu[i]-1));
    }
    if(difference>1e-10||!cr.componentwise_error_envelope_verified||!gr.componentwise_error_envelope_verified)goto finish;
    printf("SPATIAL_SELFTEST_PASS shells=%d cells=%d cpu_gpu_relative=%.17g coefficients_byte_identical=1 volume_preserved=1 centers_preserved=1 invalid_rejected=1\n",g.ns,g.ns*nb,difference);rc=0;
finish:
    free(c);free(e);free(f);free(cpu);free(gpu);free(ce);free(ge);return rc;
}

int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--selftest"))return small_test()?1:0;
    if(argc!=5)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFGENState cs={0};SplitGrid grid={0};
    double *capture[6]={0},*old=NULL,*material=NULL,*chi=NULL,*es=NULL,*eta=NULL,*j=NULL,*error=NULL,*centers=NULL;
    FILE *fp=NULL;int rc=-1,it=-1;
    REQUIRE(!getenv("LUMINA_CMF_FINE_LINE_EPS")&&!getenv("LUMINA_INNER_BB_SCALE"));
    REQUIRE(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    REQUIRE(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&it));
    int ns=m.geometry.n_shells,nb=m.opacity.n_fine;size_t cells=(size_t)ns*nb;
    REQUIRE(ns==113&&nb==117606&&it==4&&!m.plasma.physical_history);
    size_t ni=m.atom.n_ion_pops,nl=m.nlte.n_nlte_levels_total,nm=(2+ni+nl)*(size_t)ns;
    material=malloc(nm*sizeof(double));old=malloc(cells*sizeof(double));centers=malloc(cells*sizeof(double));
    REQUIRE(material&&old&&centers);
    memcpy(material,m.plasma.T_e,ns*8);memcpy(material+ns,m.plasma.n_electron,ns*8);
    memcpy(material+2*ns,m.atom.ion_number_density,ni*ns*8);
    memcpy(material+(2+ni)*ns,m.nlte.nlte_level_populations,nl*ns*8);
    memcpy(old,m.opacity.jnu_fine,cells*8);
    REQUIRE(!snapshot(&m,old,"stored_bf.csv",it));
    for(int k=0;k<6;k++){capture[k]=malloc((k==5?(size_t)nb:cells)*sizeof(double));REQUIRE(capture[k]);}
    w5_cont_selftest_arm_capture(capture[0],capture[1],capture[2],capture[3],capture[4],capture[5],cells);
    REQUIRE(!cmfgen_init(&cs,&m.geometry));
    REQUIRE(!cmfgen_initial_radiation_publish(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,&m.nlte,&m.atom,m.plasma.T_e[0],0));
    int captured_ns=0,captured_nb=0;
    REQUIRE(w5_cont_selftest_capture_shape(&captured_ns,&captured_nb)&&captured_ns==ns&&captured_nb==nb);
    REQUIRE(!memcmp(capture[5],m.opacity.nu_fine,(size_t)nb*8));
    REQUIRE(!snapshot(&m,m.opacity.jnu_fine,"baseline_bf.csv",it));
    /* Common Gaussian의 line_eps=1에서 실제 조립과 동일한 순서다. */
    for(size_t i=0;i<cells;i++){
        capture[1][i]=capture[3][i]+capture[1][i]+capture[4][i];
        capture[0][i]=capture[0][i]+capture[2][i];
        REQUIRE(isfinite(capture[1][i])&&capture[0][i]>=0&&isfinite(capture[0][i]));
    }
    REQUIRE(!split_grid(&grid,ns,m.geometry.r_inner,m.geometry.r_outer,39,44)&&grid.ns==125);
    chi=replicate(&grid,nb,capture[1]);es=replicate(&grid,nb,capture[3]);eta=replicate(&grid,nb,capture[0]);
    j=replicate(&grid,nb,m.opacity.jnu_fine);error=malloc((size_t)grid.ns*nb*sizeof(double));
    REQUIRE(chi&&es&&eta&&j&&error);
    for(int s=0;s<grid.ns;s++){
        size_t at=(size_t)s*nb,src=(size_t)grid.parent[s]*nb;
        REQUIRE(!memcmp(chi+at,capture[1]+src,nb*8)&&!memcmp(es+at,capture[3]+src,nb*8)&&!memcmp(eta+at,capture[0]+src,nb*8));
    }
    CMFMultiGPUEpochSchedule schedule={128,4096,32};CMFMultiGPUReport report={0};
    CMFMultiGPUStatus status=cmf_exact_multigpu_emissivity_solve_epoch(grid.ns,nb,m.opacity.dlognu_fine,capture[5],
        grid.ri,grid.ro,m.geometry.time_explosion,m.plasma.T_e[0],1,chi,es,eta,j,error,42,&schedule,1,768,1e-8,&report);
    printf("SPATIAL_SOLVE status=%s iterations=%d residual=%.17g absolute_bound=%.17g refinements=%zu seconds=%.9g\n",
        cmf_multigpu_status_name(status),report.iterations_used,report.final_max_relative_change,
        report.componentwise_error_upper_max,report.componentwise_error_refinement_iterations,report.total_seconds);
    REQUIRE(status==CMF_MGPU_OK&&report.componentwise_error_envelope_verified);
    for(int s=0;s<ns;s++)memcpy(centers+(size_t)s*nb,j+(size_t)grid.center[s]*nb,nb*8);
    REQUIRE(!snapshot(&m,centers,"refined_bf.csv",it));
    fp=fopen("field_change.csv","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"shell,refined_center_shell,baseline_vs_stored_integral,refined_vs_baseline_integral,refined_L1,center_error_upper_max\n")>0);
    double baseline_integral_max=0;
    for(int s=0;s<ns;s++){
        long double before=0,baseline=0,after=0,delta=0;double bound=0;
        for(int b=0;b<nb;b++){
            size_t i=(size_t)s*nb+b;double width=capture[5][b]*2*sinh(.5*m.opacity.dlognu_fine);
            before+=(long double)old[i]*width;baseline+=(long double)m.opacity.jnu_fine[i]*width;
            after+=(long double)centers[i]*width;delta+=(long double)fabs(centers[i]-m.opacity.jnu_fine[i])*width;
            bound=fmax(bound,error[(size_t)grid.center[s]*nb+b]);
        }
        REQUIRE(before>0&&baseline>0&&after>0);
        baseline_integral_max=fmax(baseline_integral_max,(double)fabsl(baseline/before-1));
        REQUIRE(fprintf(fp,"%d,%d,%.17Lg,%.17Lg,%.17Lg,%.17g\n",s,grid.center[s],baseline/before-1,after/baseline-1,delta/fmaxl(after,baseline),bound)>0);
    }
    printf("SPATIAL_BASELINE max_integral_relative_to_stored=%.17g\n",baseline_integral_max);
    REQUIRE(baseline_integral_max<1e-6);
    REQUIRE(!memcmp(material,m.plasma.T_e,ns*8)&&!memcmp(material+ns,m.plasma.n_electron,ns*8));
    REQUIRE(!memcmp(material+2*ns,m.atom.ion_number_density,ni*ns*8));
    REQUIRE(!memcmp(material+(2+ni)*ns,m.nlte.nlte_level_populations,nl*ns*8));
    puts("SPATIAL_FIELD_DONE coefficients_byte_identical=1 material_byte_identical=1 full_SE_solved=0 physical_history=0 final_goal=NOT_JUDGED");rc=0;
done:
    if(fp&&fclose(fp))rc=-1;
    for(int k=0;k<6;k++)free(capture[k]);
    free(old);free(material);free(chi);free(es);free(eta);free(j);free(error);free(centers);
    cmfgen_free(&cs);cmf_toy_model_free(&m);return rc?1:0;
}
