/* 실제 저장 물질의 수송 계수만 보존한다. RTE/새 물리해를 계산하지 않는다. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include "cmf_exact_sliding.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
static int captured;
CMFExactStatus __wrap_cmf_exact_emissivity_solve(
 int ns,int nb,double dlog,const double *nu,const double *ri,const double *ro,
 double time,double temperature,double boundary_scale,const double *chi,
 const double *scatter,const double *eta,double *J,double *error_upper,
 size_t refinements,int iterations,double tolerance,CMFExactMode mode,CMFExactReport *report){
    (void)eta;(void)J;(void)error_upper;(void)refinements;
    (void)iterations;(void)tolerance;(void)mode;
    if(report)report->status=CMF_EXACT_INVALID_INPUT;
    if(captured||ns!=113||nb!=303316)return CMF_EXACT_INVALID_INPUT;
    size_t cells=(size_t)ns*nb,negative=0,minimum=0;
    for(size_t i=0;i<cells;i++){
        if(!isfinite(chi[i])||!isfinite(scatter[i])||scatter[i]<0)
            return CMF_EXACT_INVALID_INPUT;
        if(chi[i]<0)negative++;
        if(chi[i]<chi[minimum])minimum=i;
    }
    FILE *f=fopen("operator.bin","wx");if(!f)return CMF_EXACT_INVALID_INPUT;
    uint64_t shape[2]={(uint64_t)ns,(uint64_t)nb};
    double parameters[4]={dlog,time,temperature,boundary_scale};
    int ok=fwrite("CMFOP01",1,8,f)==8&&fwrite(shape,8,2,f)==2&&
        fwrite(parameters,8,4,f)==4&&fwrite(nu,8,nb,f)==(size_t)nb&&
        fwrite(ri,8,ns,f)==(size_t)ns&&fwrite(ro,8,ns,f)==(size_t)ns&&
        fwrite(chi,8,cells,f)==cells&&fwrite(scatter,8,cells,f)==cells;
    if(fclose(f))ok=0;
    if(ok){
        captured=1;
        printf("OPERATOR_CAPTURE ns=%d nb=%d negative_total_cells=%zu min_shell=%zu min_bin=%zu min_chi=%.17g min_scatter=%.17g\n",
            ns,nb,negative,minimum/(size_t)nb,minimum%(size_t)nb,chi[minimum],scatter[minimum]);
    }
    return CMF_EXACT_INVALID_INPUT;
}
int main(int argc,char **argv){
    if(argc!=5)return 2;
    CMFToyModel m={0};CMFGENState cs={0};int rc=1,iteration=-1;
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    if(cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass)||
       cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration)||
       iteration!=0||m.plasma.n_shells!=113||m.opacity.n_fine!=303316||
       m.plasma.physical_history||!m.plasma.initial_steady_nlte||
       cmfgen_init(&cs,&m.geometry))goto done;
    uint64_t generation=m.nlte.radiation_field.field.generation.computed_generation;
    int publish=cmfgen_initial_radiation_publish(&cs,&m.geometry,&m.opacity,&m.bf,
        &m.plasma,&m.nlte,&m.atom,m.plasma.T_e[0],0);
    if(captured&&publish!=0&&m.nlte.radiation_field.field.generation.computed_generation==generation){
        puts("OPERATOR_CAPTURE_DONE intentional_formal_stop=1 new_RTE=0 physical_history=0");rc=0;
    }
done:
    cmfgen_free(&cs);cmf_toy_model_free(&m);return rc;
}
