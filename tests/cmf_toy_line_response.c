/* 실제 행렬과 선율의 작은 진단 사본. 생산 소스/풀이/입력은 바꾸지 않는다. */
#include "cmf_toy_model.h"
#include <stdio.h>
#include <stdint.h>
static int capture_Z=0;
static int line_response_capture_prepare(CMFToyModel *m,int e,int s){
    AtomicData *a=&m->atom;NLTEConfig *n=&m->nlte;int ns=m->plasma.n_shells;
    int z=a->element_Z[e],first=-1;
    for(int ip=0;ip<n->n_nlte_ions;ip++)if(n->nlte_Z[ip]==z){first=ip;break;}
    if(first<0||!n->nlte_line_global)return -1;
    int sl0=n->nlte_ion_super_offset[first];char path[80];
    snprintf(path,sizeof(path),"levels_Z%d.csv",z);FILE *f=fopen(path,"wx");if(!f)return -1;
    fprintf(f,"global_level,matrix_index,charge,within,energy_eV,g\n");
    for(int fl=0;fl<n->n_nlte_levels_total;fl++){
        int gl=n->nlte_to_global_level[fl];if(a->level_Z[gl]!=z)continue;
        fprintf(f,"%d,%d,%d,%.17g,%.17g,%d\n",gl,n->fl_to_super[fl]-sl0,a->level_ion[gl],
                n->within_sl_frac[(size_t)fl*ns+s],a->level_energy_eV[gl],a->level_g[gl]);
    }
    if(fclose(f))return -1;
    snprintf(path,sizeof(path),"lines_Z%d.csv",z);f=fopen(path,"wx");if(!f)return -1;
    fprintf(f,"line,charge,lower_global,upper_global,lower_matrix,upper_matrix,w_lower,w_upper,nu,Jbar,B_lu,B_ul,A_ul\n");
    for(size_t q=0;q<n->line_view.n_lines;q++){
        int l=(int)n->line_view.line_id[q];if(a->line_atomic_number[l]!=z)continue;
        int gl=n->nlte_line_global[2*(size_t)l],gu=n->nlte_line_global[2*(size_t)l+1];
        if(gl<0||gu<0){fclose(f);return -1;}
        int fl=n->global_to_nlte_level[gl],fu=n->global_to_nlte_level[gu];
        if(fl<0||fu<0){fclose(f);return -1;}
        fprintf(f,"%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            l,a->line_ion_number[l],gl,gu,n->fl_to_super[fl]-sl0,n->fl_to_super[fu]-sl0,
            n->within_sl_frac[(size_t)fl*ns+s],n->within_sl_frac[(size_t)fu*ns+s],
            a->line_nu[l],n->line_view.jbar[q*ns+s],a->line_B_lu[l],a->line_B_ul[l],a->line_A_ul[l]);
    }
    if(fclose(f))return -1;
    capture_Z=z;return 0;
}
PopulationStatus __real_population_generator_stationary_gth(const double *,size_t,double,double *,PopulationGeneratorSolveDiagnostic *);
PopulationStatus __wrap_population_generator_stationary_gth(const double *A,size_t n,double total,double *x,PopulationGeneratorSolveDiagnostic *d){
    PopulationStatus status=__real_population_generator_stationary_gth(A,n,total,x,d);
    if(capture_Z&&status==POP_OK){
        char path[80];snprintf(path,sizeof(path),"matrix_Z%d.bin",capture_Z);
        FILE *f=fopen(path,"wx");if(!f)return POP_SOLVE_FAILED;
        uint64_t size=n;int ok=fwrite("SLMAT01",1,8,f)==8&&fwrite(&size,8,1,f)==1&&
            fwrite(&total,8,1,f)==1&&fwrite(A,8,n*n,f)==n*n&&fwrite(x,8,n,f)==n;
        if(fclose(f))ok=0;
        if(!ok)return POP_SOLVE_FAILED;
        printf("LINE_RESPONSE_CAPTURE Z=%d N=%zu population_solver_unchanged=1\n",capture_Z,n);
        capture_Z=0;
    }
    return status;
}
#define LUMINA_LINE_RESPONSE_CAPTURE 1
#include "cmf_toy_superlevel_response.c"
